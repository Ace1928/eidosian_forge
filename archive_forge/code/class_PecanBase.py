from inspect import Arguments
from itertools import chain, tee
from mimetypes import guess_type, add_type
from os.path import splitext
import logging
import operator
import sys
import types
from webob import (Request as WebObRequest, Response as WebObResponse, exc,
from webob.multidict import NestedMultiDict
from .compat import urlparse, izip, is_bound_method as ismethod
from .jsonify import encode as dumps
from .secure import handle_security
from .templating import RendererFactory
from .routing import lookup_controller, NonCanonicalPath
from .util import _cfg, getargspec
from .middleware.recursive import ForwardRequestException
class PecanBase(object):
    SIMPLEST_CONTENT_TYPES = (['text/html'], ['text/plain'])

    def __init__(self, root, default_renderer='mako', template_path='templates', hooks=lambda: [], custom_renderers={}, extra_template_vars={}, force_canonical=True, guess_content_type_from_ext=True, context_local_factory=None, request_cls=Request, response_cls=Response, **kw):
        if isinstance(root, str):
            root = self.__translate_root__(root)
        self.root = root
        self.request_cls = request_cls
        self.response_cls = response_cls
        self.renderers = RendererFactory(custom_renderers, extra_template_vars)
        self.default_renderer = default_renderer
        if callable(hooks):
            hooks = hooks()
        self.hooks = list(sorted(hooks, key=operator.attrgetter('priority')))
        self.template_path = template_path
        self.force_canonical = force_canonical
        self.guess_content_type_from_ext = guess_content_type_from_ext

    def __translate_root__(self, item):
        """
        Creates a root controller instance from a string root, e.g.,

        > __translate_root__("myproject.controllers.RootController")
        myproject.controllers.RootController()

        :param item: The string to the item
        """
        if '.' in item:
            parts = item.split('.')
            name = '.'.join(parts[:-1])
            fromlist = parts[-1:]
            module = __import__(name, fromlist=fromlist)
            kallable = getattr(module, parts[-1])
            msg = '%s does not represent a callable class or function.'
            if not callable(kallable):
                raise TypeError(msg % item)
            return kallable()
        raise ImportError('No item named %s' % item)

    def route(self, req, node, path):
        """
        Looks up a controller from a node based upon the specified path.

        :param node: The node, such as a root controller object.
        :param path: The path to look up on this node.
        """
        path = path.split('/')[1:]
        try:
            node, remainder = lookup_controller(node, path, req)
            return (node, remainder)
        except NonCanonicalPath as e:
            if self.force_canonical and (not _cfg(e.controller).get('accept_noncanonical', False)):
                if req.method == 'POST':
                    raise RuntimeError("You have POSTed to a URL '%s' which requires a slash. Most browsers will not maintain POST data when redirected. Please update your code to POST to '%s/' or set force_canonical to False" % (req.pecan['routing_path'], req.pecan['routing_path']))
                redirect(code=302, add_slash=True, request=req)
            return (e.controller, e.remainder)

    def determine_hooks(self, controller=None):
        """
        Determines the hooks to be run, in which order.

        :param controller: If specified, includes hooks for a specific
                           controller.
        """
        controller_hooks = []
        if controller:
            controller_hooks = _cfg(controller).get('hooks', [])
            if controller_hooks:
                return list(sorted(chain(controller_hooks, self.hooks), key=operator.attrgetter('priority')))
        return self.hooks

    def handle_hooks(self, hooks, hook_type, *args):
        """
        Processes hooks of the specified type.

        :param hook_type: The type of hook, including ``before``, ``after``,
                          ``on_error``, and ``on_route``.
        :param \\*args: Arguments to pass to the hooks.
        """
        if hook_type not in ['before', 'on_route']:
            hooks = reversed(hooks)
        for hook in hooks:
            result = getattr(hook, hook_type)(*args)
            if hook_type == 'on_error' and isinstance(result, WebObResponse):
                return result

    def get_args(self, state, all_params, remainder, argspec, im_self):
        """
        Determines the arguments for a controller based upon parameters
        passed the argument specification for the controller.
        """
        args = []
        varargs = []
        kwargs = dict()
        valid_args = argspec.args[:]
        if ismethod(state.controller) or im_self:
            valid_args.pop(0)
        pecan_state = state.request.pecan
        remainder = [x for x in remainder if x]
        if im_self is not None:
            args.append(im_self)
        if 'routing_args' in pecan_state:
            remainder = pecan_state['routing_args'] + list(remainder)
            del pecan_state['routing_args']
        if valid_args and remainder:
            args.extend(remainder[:len(valid_args)])
            remainder = remainder[len(valid_args):]
            valid_args = valid_args[len(args):]
        if [i for i in remainder if i]:
            if not argspec[1]:
                abort(404)
            varargs.extend(remainder)
        if argspec[3]:
            defaults = dict(izip(argspec[0][-len(argspec[3]):], argspec[3]))
        else:
            defaults = dict()
        for name in valid_args:
            if name in all_params:
                args.append(all_params.pop(name))
            elif name in defaults:
                args.append(defaults[name])
            else:
                break
        if argspec[2]:
            for name, value in iter(all_params.items()):
                if name not in argspec[0]:
                    kwargs[name] = value
        return (args, varargs, kwargs)

    def render(self, template, namespace):
        if template == 'json':
            renderer = self.renderers.get('json', self.template_path)
        elif ':' in template:
            renderer_name, template = template.split(':', 1)
            renderer = self.renderers.get(renderer_name, self.template_path)
            if renderer is None:
                raise RuntimeError('support for "%s" was not found; ' % renderer_name + 'supported template engines are %s' % self.renderers.keys())
        else:
            renderer = self.renderers.get(self.default_renderer, self.template_path)
        return renderer.render(template, namespace)

    def find_controller(self, state):
        """
        The main request handler for Pecan applications.
        """
        req = state.request
        pecan_state = req.pecan
        pecan_state['routing_path'] = path = req.path_info
        self.handle_hooks(self.hooks, 'on_route', state)
        pecan_state['extension'] = None
        if self.guess_content_type_from_ext and (not pecan_state['content_type']) and ('.' in path):
            _, extension = splitext(path.rstrip('/'))
            potential_type = guess_type('x' + extension)[0]
            if extension and potential_type is not None:
                path = ''.join(path.rsplit(extension, 1))
                pecan_state['extension'] = extension
                pecan_state['content_type'] = potential_type
        controller, remainder = self.route(req, self.root, path)
        cfg = _cfg(controller)
        if cfg.get('generic_handler'):
            raise exc.HTTPNotFound
        im_self = None
        if cfg.get('generic'):
            im_self = controller.__self__
            handlers = cfg['generic_handlers']
            controller = handlers.get(req.method, handlers['DEFAULT'])
            handle_security(controller, im_self)
            cfg = _cfg(controller)
        state.controller = controller
        content_types = cfg.get('content_types', {})
        if not pecan_state['content_type']:
            accept = getattr(req.accept, 'header_value', '*/*') or '*/*'
            if accept == '*/*' or (accept.startswith('text/html,') and list(content_types.keys()) in self.SIMPLEST_CONTENT_TYPES):
                pecan_state['content_type'] = cfg.get('content_type', 'text/html')
            else:
                best_default = None
                accept_header = acceptparse.create_accept_header(accept)
                offers = accept_header.acceptable_offers(list(content_types.keys()))
                if offers:
                    best_default = offers[0][0]
                else:
                    for k in content_types.keys():
                        if accept.startswith(k):
                            best_default = k
                            break
                if best_default is None:
                    msg = "Controller '%s' defined does not support " + "content_type '%s'. Supported type(s): %s"
                    logger.error(msg % (controller.__name__, pecan_state['content_type'], content_types.keys()))
                    raise exc.HTTPNotAcceptable()
                pecan_state['content_type'] = best_default
        elif cfg.get('content_type') is not None and pecan_state['content_type'] not in content_types:
            msg = "Controller '%s' defined does not support content_type " + "'%s'. Supported type(s): %s"
            logger.error(msg % (controller.__name__, pecan_state['content_type'], content_types.keys()))
            raise exc.HTTPNotFound
        if req.method == 'GET':
            params = req.GET
        elif req.content_type in ('application/json', 'application/javascript'):
            try:
                if not isinstance(req.json, dict):
                    raise TypeError('%s is not a dict' % req.json)
                params = NestedMultiDict(req.GET, req.json)
            except (TypeError, ValueError):
                params = req.params
        else:
            params = req.params
        args, varargs, kwargs = self.get_args(state, params.mixed(), remainder, cfg['argspec'], im_self)
        state.arguments = Arguments(args, varargs, kwargs)
        self.handle_hooks(self.determine_hooks(controller), 'before', state)
        return (controller, args + varargs, kwargs)

    def invoke_controller(self, controller, args, kwargs, state):
        """
        The main request handler for Pecan applications.
        """
        cfg = _cfg(controller)
        content_types = cfg.get('content_types', {})
        req = state.request
        resp = state.response
        pecan_state = req.pecan
        argspec = getargspec(controller)
        keys = kwargs.keys()
        for key in keys:
            if key not in argspec.args and (not argspec.keywords):
                kwargs.pop(key)
        result = controller(*args, **kwargs)
        if result is response:
            return
        elif isinstance(result, WebObResponse):
            state.response = result
            return
        raw_namespace = result
        template = content_types.get(pecan_state['content_type'])
        template = pecan_state.get('override_template', template)
        if template is None and cfg['explicit_content_type'] is False:
            if self.default_renderer == 'json':
                template = 'json'
        pecan_state['content_type'] = pecan_state.get('override_content_type', pecan_state['content_type'])
        if template:
            if template == 'json':
                pecan_state['content_type'] = 'application/json'
            result = self.render(template, result)
        if req.environ.get('paste.testing'):
            testing_variables = req.environ['paste.testing_variables']
            testing_variables['namespace'] = raw_namespace
            testing_variables['template_name'] = template
            testing_variables['controller_output'] = result
        if result and isinstance(result, str):
            resp.text = result
        elif result:
            resp.body = result
        if pecan_state['content_type']:
            resp.content_type = pecan_state['content_type']

    def _handle_empty_response_body(self, state):
        if state.response.status_int == 200:
            if isinstance(state.response.app_iter, types.GeneratorType):
                a, b = tee(state.response.app_iter)
                try:
                    next(a)
                except StopIteration:
                    state.response.status = 204
                finally:
                    state.response.app_iter = b
            else:
                text = None
                if state.response.charset:
                    try:
                        text = state.response.text
                    except UnicodeDecodeError:
                        pass
                if not any((state.response.body, text)):
                    state.response.status = 204
        if state.response.status_int in (204, 304):
            state.response.content_type = None

    def __call__(self, environ, start_response):
        """
        Implements the WSGI specification for Pecan applications, utilizing
        ``WebOb``.
        """
        req = self.request_cls(environ)
        resp = self.response_cls()
        state = RoutingState(req, resp, self)
        environ['pecan.locals'] = {'request': req, 'response': resp}
        controller = None
        internal_redirect = False
        try:
            req.context = environ.get('pecan.recursive.context', {})
            req.pecan = dict(content_type=None)
            controller, args, kwargs = self.find_controller(state)
            self.invoke_controller(controller, args, kwargs, state)
        except Exception as e:
            if isinstance(e, exc.HTTPException):
                accept_header = acceptparse.create_accept_header(getattr(req.accept, 'header_value', '*/*') or '*/*')
                offers = accept_header.acceptable_offers(('text/plain', 'text/html', 'application/json'))
                best_match = offers[0][0] if offers else None
                state.response = e
                if best_match == 'application/json':
                    json_body = dumps({'code': e.status_int, 'title': e.title, 'description': e.detail})
                    if isinstance(json_body, str):
                        e.text = json_body
                    else:
                        e.body = json_body
                    state.response.content_type = best_match
                environ['pecan.original_exception'] = e
            internal_redirect = isinstance(e, ForwardRequestException)
            on_error_result = None
            if not internal_redirect:
                on_error_result = self.handle_hooks(self.determine_hooks(state.controller), 'on_error', state, e)
            if isinstance(on_error_result, WebObResponse):
                state.response = on_error_result
            elif not isinstance(e, exc.HTTPException):
                raise
            if isinstance(e, exc.HTTPMethodNotAllowed) and controller:
                allowed_methods = _cfg(controller).get('allowed_methods', [])
                if allowed_methods:
                    state.response.allow = sorted(allowed_methods)
        finally:
            if not internal_redirect:
                self.handle_hooks(self.determine_hooks(state.controller), 'after', state)
        self._handle_empty_response_body(state)
        return state.response(environ, start_response)