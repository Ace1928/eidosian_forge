from xmlrpc.client import Fault, dumps, loads, gzip_encode, gzip_decode
from http.server import BaseHTTPRequestHandler
from functools import partial
from inspect import signature
import html
import http.server
import socketserver
import sys
import os
import re
import pydoc
import traceback
class SimpleXMLRPCDispatcher:
    """Mix-in class that dispatches XML-RPC requests.

    This class is used to register XML-RPC method handlers
    and then to dispatch them. This class doesn't need to be
    instanced directly when used by SimpleXMLRPCServer but it
    can be instanced when used by the MultiPathXMLRPCServer
    """

    def __init__(self, allow_none=False, encoding=None, use_builtin_types=False):
        self.funcs = {}
        self.instance = None
        self.allow_none = allow_none
        self.encoding = encoding or 'utf-8'
        self.use_builtin_types = use_builtin_types

    def register_instance(self, instance, allow_dotted_names=False):
        """Registers an instance to respond to XML-RPC requests.

        Only one instance can be installed at a time.

        If the registered instance has a _dispatch method then that
        method will be called with the name of the XML-RPC method and
        its parameters as a tuple
        e.g. instance._dispatch('add',(2,3))

        If the registered instance does not have a _dispatch method
        then the instance will be searched to find a matching method
        and, if found, will be called. Methods beginning with an '_'
        are considered private and will not be called by
        SimpleXMLRPCServer.

        If a registered function matches an XML-RPC request, then it
        will be called instead of the registered instance.

        If the optional allow_dotted_names argument is true and the
        instance does not have a _dispatch method, method names
        containing dots are supported and resolved, as long as none of
        the name segments start with an '_'.

            *** SECURITY WARNING: ***

            Enabling the allow_dotted_names options allows intruders
            to access your module's global variables and may allow
            intruders to execute arbitrary code on your machine.  Only
            use this option on a secure, closed network.

        """
        self.instance = instance
        self.allow_dotted_names = allow_dotted_names

    def register_function(self, function=None, name=None):
        """Registers a function to respond to XML-RPC requests.

        The optional name argument can be used to set a Unicode name
        for the function.
        """
        if function is None:
            return partial(self.register_function, name=name)
        if name is None:
            name = function.__name__
        self.funcs[name] = function
        return function

    def register_introspection_functions(self):
        """Registers the XML-RPC introspection methods in the system
        namespace.

        see http://xmlrpc.usefulinc.com/doc/reserved.html
        """
        self.funcs.update({'system.listMethods': self.system_listMethods, 'system.methodSignature': self.system_methodSignature, 'system.methodHelp': self.system_methodHelp})

    def register_multicall_functions(self):
        """Registers the XML-RPC multicall method in the system
        namespace.

        see http://www.xmlrpc.com/discuss/msgReader$1208"""
        self.funcs.update({'system.multicall': self.system_multicall})

    def _marshaled_dispatch(self, data, dispatch_method=None, path=None):
        """Dispatches an XML-RPC method from marshalled (XML) data.

        XML-RPC methods are dispatched from the marshalled (XML) data
        using the _dispatch method and the result is returned as
        marshalled data. For backwards compatibility, a dispatch
        function can be provided as an argument (see comment in
        SimpleXMLRPCRequestHandler.do_POST) but overriding the
        existing method through subclassing is the preferred means
        of changing method dispatch behavior.
        """
        try:
            params, method = loads(data, use_builtin_types=self.use_builtin_types)
            if dispatch_method is not None:
                response = dispatch_method(method, params)
            else:
                response = self._dispatch(method, params)
            response = (response,)
            response = dumps(response, methodresponse=1, allow_none=self.allow_none, encoding=self.encoding)
        except Fault as fault:
            response = dumps(fault, allow_none=self.allow_none, encoding=self.encoding)
        except BaseException as exc:
            response = dumps(Fault(1, '%s:%s' % (type(exc), exc)), encoding=self.encoding, allow_none=self.allow_none)
        return response.encode(self.encoding, 'xmlcharrefreplace')

    def system_listMethods(self):
        """system.listMethods() => ['add', 'subtract', 'multiple']

        Returns a list of the methods supported by the server."""
        methods = set(self.funcs.keys())
        if self.instance is not None:
            if hasattr(self.instance, '_listMethods'):
                methods |= set(self.instance._listMethods())
            elif not hasattr(self.instance, '_dispatch'):
                methods |= set(list_public_methods(self.instance))
        return sorted(methods)

    def system_methodSignature(self, method_name):
        """system.methodSignature('add') => [double, int, int]

        Returns a list describing the signature of the method. In the
        above example, the add method takes two integers as arguments
        and returns a double result.

        This server does NOT support system.methodSignature."""
        return 'signatures not supported'

    def system_methodHelp(self, method_name):
        """system.methodHelp('add') => "Adds two integers together"

        Returns a string containing documentation for the specified method."""
        method = None
        if method_name in self.funcs:
            method = self.funcs[method_name]
        elif self.instance is not None:
            if hasattr(self.instance, '_methodHelp'):
                return self.instance._methodHelp(method_name)
            elif not hasattr(self.instance, '_dispatch'):
                try:
                    method = resolve_dotted_attribute(self.instance, method_name, self.allow_dotted_names)
                except AttributeError:
                    pass
        if method is None:
            return ''
        else:
            return pydoc.getdoc(method)

    def system_multicall(self, call_list):
        """system.multicall([{'methodName': 'add', 'params': [2, 2]}, ...]) => [[4], ...]

        Allows the caller to package multiple XML-RPC calls into a single
        request.

        See http://www.xmlrpc.com/discuss/msgReader$1208
        """
        results = []
        for call in call_list:
            method_name = call['methodName']
            params = call['params']
            try:
                results.append([self._dispatch(method_name, params)])
            except Fault as fault:
                results.append({'faultCode': fault.faultCode, 'faultString': fault.faultString})
            except BaseException as exc:
                results.append({'faultCode': 1, 'faultString': '%s:%s' % (type(exc), exc)})
        return results

    def _dispatch(self, method, params):
        """Dispatches the XML-RPC method.

        XML-RPC calls are forwarded to a registered function that
        matches the called XML-RPC method name. If no such function
        exists then the call is forwarded to the registered instance,
        if available.

        If the registered instance has a _dispatch method then that
        method will be called with the name of the XML-RPC method and
        its parameters as a tuple
        e.g. instance._dispatch('add',(2,3))

        If the registered instance does not have a _dispatch method
        then the instance will be searched to find a matching method
        and, if found, will be called.

        Methods beginning with an '_' are considered private and will
        not be called.
        """
        try:
            func = self.funcs[method]
        except KeyError:
            pass
        else:
            if func is not None:
                return func(*params)
            raise Exception('method "%s" is not supported' % method)
        if self.instance is not None:
            if hasattr(self.instance, '_dispatch'):
                return self.instance._dispatch(method, params)
            try:
                func = resolve_dotted_attribute(self.instance, method, self.allow_dotted_names)
            except AttributeError:
                pass
            else:
                if func is not None:
                    return func(*params)
        raise Exception('method "%s" is not supported' % method)