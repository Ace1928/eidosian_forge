import os
import sys
import importlib.util as imputil
import mimetypes
from paste import request
from paste import fileapp
from paste.util import import_string
from paste import httpexceptions
from .httpheaders import ETAG
from paste.util import converters
class URLParser(object):
    """
    WSGI middleware

    Application dispatching, based on URL.  An instance of `URLParser` is
    an application that loads and delegates to other applications.  It
    looks for files in its directory that match the first part of
    PATH_INFO; these may have an extension, but are not required to have
    one, in which case the available files are searched to find the
    appropriate file.  If it is ambiguous, a 404 is returned and an error
    logged.

    By default there is a constructor for .py files that loads the module,
    and looks for an attribute ``application``, which is a ready
    application object, or an attribute that matches the module name,
    which is a factory for building applications, and is called with no
    arguments.

    URLParser will also look in __init__.py for special overrides.
    These overrides are:

    ``urlparser_hook(environ)``
        This can modify the environment.  Its return value is ignored,
        and it cannot be used to change the response in any way.  You
        *can* use this, for example, to manipulate SCRIPT_NAME/PATH_INFO
        (try to keep them consistent with the original URL -- but
        consuming PATH_INFO and moving that to SCRIPT_NAME is ok).

    ``urlparser_wrap(environ, start_response, app)``:
        After URLParser finds the application, it calls this function
        (if present).  If this function doesn't call
        ``app(environ, start_response)`` then the application won't be
        called at all!  This can be used to allocate resources (with
        ``try:finally:``) or otherwise filter the output of the
        application.

    ``not_found_hook(environ, start_response)``:
        If no file can be found (*in this directory*) to match the
        request, then this WSGI application will be called.  You can
        use this to change the URL and pass the request back to
        URLParser again, or on to some other application.  This
        doesn't catch all ``404 Not Found`` responses, just missing
        files.

    ``application(environ, start_response)``:
        This basically overrides URLParser completely, and the given
        application is used for all requests.  ``urlparser_wrap`` and
        ``urlparser_hook`` are still called, but the filesystem isn't
        searched in any way.
    """
    parsers_by_directory = {}
    init_module = NoDefault
    global_constructors = {}

    def __init__(self, global_conf, directory, base_python_name, index_names=NoDefault, hide_extensions=NoDefault, ignore_extensions=NoDefault, constructors=None, **constructor_conf):
        """
        Create a URLParser object that looks at `directory`.
        `base_python_name` is the package that this directory
        represents, thus any Python modules in this directory will
        be given names under this package.
        """
        if global_conf:
            import warnings
            warnings.warn('The global_conf argument to URLParser is deprecated; either pass in None or {}, or use make_url_parser', DeprecationWarning)
        else:
            global_conf = {}
        if os.path.sep != '/':
            directory = directory.replace(os.path.sep, '/')
        self.directory = directory
        self.base_python_name = base_python_name
        if index_names is NoDefault:
            index_names = global_conf.get('index_names', ('index', 'Index', 'main', 'Main'))
        self.index_names = converters.aslist(index_names)
        if hide_extensions is NoDefault:
            hide_extensions = global_conf.get('hide_extensions', ('.pyc', '.bak', '.py~', '.pyo'))
        self.hide_extensions = converters.aslist(hide_extensions)
        if ignore_extensions is NoDefault:
            ignore_extensions = global_conf.get('ignore_extensions', ())
        self.ignore_extensions = converters.aslist(ignore_extensions)
        self.constructors = self.global_constructors.copy()
        if constructors:
            self.constructors.update(constructors)
        for name, value in constructor_conf.items():
            if not name.startswith('constructor '):
                raise ValueError("Only extra configuration keys allowed are 'constructor .ext = import_expr'; you gave %r (=%r)" % (name, value))
            ext = name[len('constructor '):].strip()
            if isinstance(value, str):
                value = import_string.eval_import(value)
            self.constructors[ext] = value

    def __call__(self, environ, start_response):
        environ['paste.urlparser.base_python_name'] = self.base_python_name
        if self.init_module is NoDefault:
            self.init_module = self.find_init_module(environ)
        path_info = environ.get('PATH_INFO', '')
        if not path_info:
            return self.add_slash(environ, start_response)
        if self.init_module and getattr(self.init_module, 'urlparser_hook', None):
            self.init_module.urlparser_hook(environ)
        orig_path_info = environ['PATH_INFO']
        orig_script_name = environ['SCRIPT_NAME']
        application, filename = self.find_application(environ)
        if not application:
            if self.init_module and getattr(self.init_module, 'not_found_hook', None) and (environ.get('paste.urlparser.not_found_parser') is not self):
                not_found_hook = self.init_module.not_found_hook
                environ['paste.urlparser.not_found_parser'] = self
                environ['PATH_INFO'] = orig_path_info
                environ['SCRIPT_NAME'] = orig_script_name
                return not_found_hook(environ, start_response)
            if filename is None:
                name, rest_of_path = request.path_info_split(environ['PATH_INFO'])
                if not name:
                    name = 'one of %s' % ', '.join(self.index_names or ['(no index_names defined)'])
                return self.not_found(environ, start_response, 'Tried to load %s from directory %s' % (name, self.directory))
            else:
                environ['wsgi.errors'].write('Found resource %s, but could not construct application\n' % filename)
                return self.not_found(environ, start_response, 'Tried to load %s from directory %s' % (filename, self.directory))
        if self.init_module and getattr(self.init_module, 'urlparser_wrap', None):
            return self.init_module.urlparser_wrap(environ, start_response, application)
        else:
            return application(environ, start_response)

    def find_application(self, environ):
        if self.init_module and getattr(self.init_module, 'application', None) and (not environ.get('paste.urlparser.init_application') == environ['SCRIPT_NAME']):
            environ['paste.urlparser.init_application'] = environ['SCRIPT_NAME']
            return (self.init_module.application, None)
        name, rest_of_path = request.path_info_split(environ['PATH_INFO'])
        environ['PATH_INFO'] = rest_of_path
        if name is not None:
            environ['SCRIPT_NAME'] = environ.get('SCRIPT_NAME', '') + '/' + name
        if not name:
            names = self.index_names
            for index_name in names:
                filename = self.find_file(environ, index_name)
                if filename:
                    break
            else:
                filename = None
        else:
            filename = self.find_file(environ, name)
        if filename is None:
            return (None, filename)
        else:
            return (self.get_application(environ, filename), filename)

    def not_found(self, environ, start_response, debug_message=None):
        exc = httpexceptions.HTTPNotFound('The resource at %s could not be found' % request.construct_url(environ), comment=debug_message)
        return exc.wsgi_application(environ, start_response)

    def add_slash(self, environ, start_response):
        """
        This happens when you try to get to a directory
        without a trailing /
        """
        url = request.construct_url(environ, with_query_string=False)
        url += '/'
        if environ.get('QUERY_STRING'):
            url += '?' + environ['QUERY_STRING']
        exc = httpexceptions.HTTPMovedPermanently('The resource has moved to %s - you should be redirected automatically.' % url, headers=[('location', url)])
        return exc.wsgi_application(environ, start_response)

    def find_file(self, environ, base_filename):
        possible = []
        'Cache a few values to reduce function call overhead'
        for filename in os.listdir(self.directory):
            base, ext = os.path.splitext(filename)
            full_filename = os.path.join(self.directory, filename)
            if ext in self.hide_extensions or not base:
                continue
            if filename == base_filename:
                possible.append(full_filename)
                continue
            if ext in self.ignore_extensions:
                continue
            if base == base_filename:
                possible.append(full_filename)
        if not possible:
            return None
        if len(possible) > 1:
            if full_filename in possible:
                return full_filename
            else:
                environ['wsgi.errors'].write('Ambiguous URL: %s; matches files %s\n' % (request.construct_url(environ), ', '.join(possible)))
            return None
        return possible[0]

    def get_application(self, environ, filename):
        if os.path.isdir(filename):
            t = 'dir'
        else:
            t = os.path.splitext(filename)[1]
        constructor = self.constructors.get(t, self.constructors.get('*'))
        if constructor is None:
            return constructor
        app = constructor(self, environ, filename)
        if app is None:
            pass
        return app

    def register_constructor(cls, extension, constructor):
        """
        Register a function as a constructor.  Registered constructors
        apply to all instances of `URLParser`.

        The extension should have a leading ``.``, or the special
        extensions ``dir`` (for directories) and ``*`` (a catch-all).

        `constructor` must be a callable that takes two arguments:
        ``environ`` and ``filename``, and returns a WSGI application.
        """
        d = cls.global_constructors
        assert extension not in d, 'A constructor already exists for the extension %r (%r) when attemption to register constructor %r' % (extension, d[extension], constructor)
        d[extension] = constructor
    register_constructor = classmethod(register_constructor)

    def get_parser(self, directory, base_python_name):
        """
        Get a parser for the given directory, or create one if
        necessary.  This way parsers can be cached and reused.

        # @@: settings are inherited from the first caller
        """
        try:
            return self.parsers_by_directory[directory, base_python_name]
        except KeyError:
            parser = self.__class__({}, directory, base_python_name, index_names=self.index_names, hide_extensions=self.hide_extensions, ignore_extensions=self.ignore_extensions, constructors=self.constructors)
            self.parsers_by_directory[directory, base_python_name] = parser
            return parser

    def find_init_module(self, environ):
        filename = os.path.join(self.directory, '__init__.py')
        if not os.path.exists(filename):
            return None
        return load_module(environ, filename)

    def __repr__(self):
        return '<%s directory=%r; module=%s at %s>' % (self.__class__.__name__, self.directory, self.base_python_name, hex(abs(id(self))))