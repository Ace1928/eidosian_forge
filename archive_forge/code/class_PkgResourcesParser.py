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
class PkgResourcesParser(StaticURLParser):

    def __init__(self, egg_or_spec, resource_name, manager=None, root_resource=None):
        if pkg_resources is None:
            raise NotImplementedError('This class requires pkg_resources.')
        if isinstance(egg_or_spec, (bytes, str)):
            self.egg = pkg_resources.get_distribution(egg_or_spec)
        else:
            self.egg = egg_or_spec
        self.resource_name = resource_name
        if manager is None:
            manager = pkg_resources.ResourceManager()
        self.manager = manager
        if root_resource is None:
            root_resource = resource_name
        self.root_resource = os.path.normpath(root_resource)

    def __repr__(self):
        return '<%s for %s:%r>' % (self.__class__.__name__, self.egg.project_name, self.resource_name)

    def __call__(self, environ, start_response):
        path_info = environ.get('PATH_INFO', '')
        if not path_info:
            return self.add_slash(environ, start_response)
        if path_info == '/':
            filename = 'index.html'
        else:
            filename = request.path_info_pop(environ)
        resource = os.path.normcase(os.path.normpath(self.resource_name + '/' + filename))
        if self.root_resource is not None and (not resource.startswith(self.root_resource)):
            return self.not_found(environ, start_response)
        if not self.egg.has_resource(resource):
            return self.not_found(environ, start_response)
        if self.egg.resource_isdir(resource):
            child_root = self.root_resource is not None and self.root_resource or self.resource_name
            return self.__class__(self.egg, resource, self.manager, root_resource=child_root)(environ, start_response)
        if environ.get('PATH_INFO') and environ.get('PATH_INFO') != '/':
            return self.error_extra_path(environ, start_response)
        type, encoding = mimetypes.guess_type(resource)
        if not type:
            type = 'application/octet-stream'
        try:
            file = self.egg.get_resource_stream(self.manager, resource)
        except (IOError, OSError) as e:
            exc = httpexceptions.HTTPForbidden('You are not permitted to view this file (%s)' % e)
            return exc.wsgi_application(environ, start_response)
        start_response('200 OK', [('content-type', type)])
        return fileapp._FileIter(file)

    def not_found(self, environ, start_response, debug_message=None):
        exc = httpexceptions.HTTPNotFound('The resource at %s could not be found' % request.construct_url(environ), comment='SCRIPT_NAME=%r; PATH_INFO=%r; looking in egg:%s#%r; debug: %s' % (environ.get('SCRIPT_NAME'), environ.get('PATH_INFO'), self.egg, self.resource_name, debug_message or '(none)'))
        return exc.wsgi_application(environ, start_response)