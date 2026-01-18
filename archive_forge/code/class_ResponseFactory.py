from decimal import Decimal
from boto.compat import filter, map
class ResponseFactory(object):

    def __init__(self, scopes=None):
        self.scopes = [] if scopes is None else scopes

    def element_factory(self, name, parent):

        class DynamicElement(parent):
            _name = name
        setattr(DynamicElement, '__name__', str(name))
        return DynamicElement

    def search_scopes(self, key):
        for scope in self.scopes:
            if hasattr(scope, key):
                return getattr(scope, key)
            if hasattr(scope, '__getitem__'):
                if key in scope:
                    return scope[key]

    def find_element(self, action, suffix, parent):
        element = self.search_scopes(action + suffix)
        if element is not None:
            return element
        if action.endswith('ByNextToken'):
            element = self.search_scopes(action[:-len('ByNextToken')] + suffix)
            if element is not None:
                return self.element_factory(action + suffix, element)
        return self.element_factory(action + suffix, parent)

    def __call__(self, action, connection=None):
        response = self.find_element(action, 'Response', Response)
        if not hasattr(response, action + 'Result'):
            result = self.find_element(action, 'Result', ResponseElement)
            setattr(response, action + 'Result', Element(result))
        return response(connection=connection)