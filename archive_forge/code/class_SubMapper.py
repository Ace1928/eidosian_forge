import collections
import itertools as it
import re
import threading
from repoze.lru import LRUCache
import six
from routes import request_config
from routes.util import (
from routes.route import Route
class SubMapper(SubMapperParent):
    """Partial mapper for use with_options"""

    def __init__(self, obj, resource_name=None, collection_name=None, actions=None, formatted=None, **kwargs):
        self.kwargs = kwargs
        self.obj = obj
        self.collection_name = collection_name
        self.member = None
        self.resource_name = resource_name or getattr(obj, 'resource_name', None) or kwargs.get('controller', None) or getattr(obj, 'controller', None)
        if formatted is not None:
            self.formatted = formatted
        else:
            self.formatted = getattr(obj, 'formatted', None)
            if self.formatted is None:
                self.formatted = True
        self.add_actions(actions or [], **kwargs)

    def connect(self, routename, path=None, **kwargs):
        newkargs = {}
        _routename = routename
        _path = path
        for key, value in six.iteritems(self.kwargs):
            if key == 'path_prefix':
                if path is not None:
                    _path = ''.join((self.kwargs[key], path))
                else:
                    _path = ''.join((self.kwargs[key], routename))
            elif key == 'name_prefix':
                if path is not None:
                    _routename = ''.join((self.kwargs[key], routename))
                else:
                    _routename = None
            elif key in kwargs:
                if isinstance(value, dict):
                    newkargs[key] = dict(value, **kwargs[key])
                else:
                    newkargs[key] = kwargs[key]
            else:
                newkargs[key] = self.kwargs[key]
        for key in kwargs:
            if key not in self.kwargs:
                newkargs[key] = kwargs[key]
        newargs = (_routename, _path)
        return self.obj.connect(*newargs, **newkargs)

    def link(self, rel=None, name=None, action=None, method='GET', formatted=None, **kwargs):
        """Generates a named route for a subresource.

        Example::

            >>> from routes.util import url_for
            >>> map = Mapper(controller_scan=None)
            >>> c = map.collection('entries', 'entry')
            >>> c.link('recent', name='recent_entries')
            >>> c.member.link('ping', method='POST', formatted=True)
            >>> url_for('entries') == '/entries'
            True
            >>> url_for('recent_entries') == '/entries/recent'
            True
            >>> url_for('ping_entry', id=1) == '/entries/1/ping'
            True
            >>> url_for('ping_entry', id=1, format='xml') == '/entries/1/ping.xml'
            True

        """
        if formatted or (formatted is None and self.formatted):
            suffix = '{.format}'
        else:
            suffix = ''
        return self.connect(name or rel + '_' + self.resource_name, '/' + (rel or name) + suffix, action=action or rel or name, **_kwargs_with_conditions(kwargs, method))

    def new(self, **kwargs):
        """Generates the "new" link for a collection submapper."""
        return self.link(rel='new', **kwargs)

    def edit(self, **kwargs):
        """Generates the "edit" link for a collection member submapper."""
        return self.link(rel='edit', **kwargs)

    def action(self, name=None, action=None, method='GET', formatted=None, **kwargs):
        """Generates a named route at the base path of a submapper.

        Example::

            >>> from routes import url_for
            >>> map = Mapper(controller_scan=None)
            >>> c = map.submapper(path_prefix='/entries', controller='entry')
            >>> c.action(action='index', name='entries', formatted=True)
            >>> c.action(action='create', method='POST')
            >>> url_for(controller='entry', action='index', method='GET') == '/entries'
            True
            >>> url_for(controller='entry', action='index', method='GET', format='xml') == '/entries.xml'
            True
            >>> url_for(controller='entry', action='create', method='POST') == '/entries'
            True

        """
        if formatted or (formatted is None and self.formatted):
            suffix = '{.format}'
        else:
            suffix = ''
        return self.connect(name or action + '_' + self.resource_name, suffix, action=action or name, **_kwargs_with_conditions(kwargs, method))

    def index(self, name=None, **kwargs):
        """Generates the "index" action for a collection submapper."""
        return self.action(name=name or self.collection_name, action='index', method='GET', **kwargs)

    def show(self, name=None, **kwargs):
        """Generates the "show" action for a collection member submapper."""
        return self.action(name=name or self.resource_name, action='show', method='GET', **kwargs)

    def create(self, **kwargs):
        """Generates the "create" action for a collection submapper."""
        return self.action(action='create', method='POST', **kwargs)

    def update(self, **kwargs):
        """Generates the "update" action for a collection member submapper."""
        return self.action(action='update', method='PUT', **kwargs)

    def delete(self, **kwargs):
        """Generates the "delete" action for a collection member submapper."""
        return self.action(action='delete', method='DELETE', **kwargs)

    def add_actions(self, actions, **kwargs):
        [getattr(self, action)(**kwargs) for action in actions]

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        pass