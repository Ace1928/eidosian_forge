from __future__ import (absolute_import, division, print_function)
import ast
import datetime
import os
import pwd
import re
import time
from collections.abc import Iterator, Sequence, Mapping, MappingView, MutableMapping
from contextlib import contextmanager
from numbers import Number
from traceback import format_exc
from jinja2.exceptions import TemplateSyntaxError, UndefinedError, SecurityError
from jinja2.loaders import FileSystemLoader
from jinja2.nativetypes import NativeEnvironment
from jinja2.runtime import Context, StrictUndefined
from ansible import constants as C
from ansible.errors import (
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.common.collections import is_sequence
from ansible.plugins.loader import filter_loader, lookup_loader, test_loader
from ansible.template.native_helpers import ansible_native_concat, ansible_eval_concat, ansible_concat
from ansible.template.template import AnsibleJ2Template
from ansible.template.vars import AnsibleJ2Vars
from ansible.utils.display import Display
from ansible.utils.listify import listify_lookup_plugin_terms
from ansible.utils.native_jinja import NativeJinjaText
from ansible.utils.unsafe_proxy import to_unsafe_text, wrap_var, AnsibleUnsafeText, AnsibleUnsafeBytes, NativeJinjaUnsafeText
class Templar:
    """
    The main class for templating, with the main entry-point of template().
    """

    def __init__(self, loader, variables=None):
        self._loader = loader
        self._available_variables = {} if variables is None else variables
        self._fail_on_undefined_errors = C.DEFAULT_UNDEFINED_VAR_BEHAVIOR
        environment_class = AnsibleNativeEnvironment if C.DEFAULT_JINJA2_NATIVE else AnsibleEnvironment
        self.environment = environment_class(extensions=self._get_extensions(), loader=FileSystemLoader(loader.get_basedir() if loader else '.'))
        self.environment.template_class.environment_class = environment_class
        self.environment.globals['lookup'] = self._lookup
        self.environment.globals['query'] = self.environment.globals['q'] = self._query_lookup
        self.environment.globals['now'] = self._now_datetime
        self.environment.globals['undef'] = self._make_undefined
        self.cur_context = None
        self._compile_single_var(self.environment)
        self.jinja2_native = C.DEFAULT_JINJA2_NATIVE

    def _compile_single_var(self, env):
        self.SINGLE_VAR = re.compile('^%s\\s*(\\w*)\\s*%s$' % (env.variable_start_string, env.variable_end_string))

    def copy_with_new_env(self, environment_class=AnsibleEnvironment, **kwargs):
        """Creates a new copy of Templar with a new environment.

        :kwarg environment_class: Environment class used for creating a new environment.
        :kwarg \\*\\*kwargs: Optional arguments for the new environment that override existing
            environment attributes.

        :returns: Copy of Templar with updated environment.
        """
        new_env = object.__new__(environment_class)
        new_env.__dict__.update(self.environment.__dict__)
        new_templar = object.__new__(Templar)
        new_templar.__dict__.update(self.__dict__)
        new_templar.environment = new_env
        new_templar.jinja2_native = environment_class is AnsibleNativeEnvironment
        mapping = {'available_variables': new_templar, 'searchpath': new_env.loader}
        for key, value in kwargs.items():
            obj = mapping.get(key, new_env)
            try:
                if value is not None:
                    setattr(obj, key, value)
            except AttributeError:
                pass
        return new_templar

    def _get_extensions(self):
        """
        Return jinja2 extensions to load.

        If some extensions are set via jinja_extensions in ansible.cfg, we try
        to load them with the jinja environment.
        """
        jinja_exts = []
        if C.DEFAULT_JINJA2_EXTENSIONS:
            jinja_exts = C.DEFAULT_JINJA2_EXTENSIONS.replace(' ', '').split(',')
        return jinja_exts

    @property
    def available_variables(self):
        return self._available_variables

    @available_variables.setter
    def available_variables(self, variables):
        """
        Sets the list of template variables this Templar instance will use
        to template things, so we don't have to pass them around between
        internal methods. We also clear the template cache here, as the variables
        are being changed.
        """
        if not isinstance(variables, Mapping):
            raise AnsibleAssertionError("the type of 'variables' should be a Mapping but was a %s" % type(variables))
        self._available_variables = variables

    @contextmanager
    def set_temporary_context(self, **kwargs):
        """Context manager used to set temporary templating context, without having to worry about resetting
        original values afterward

        Use a keyword that maps to the attr you are setting. Applies to ``self.environment`` by default, to
        set context on another object, it must be in ``mapping``.
        """
        mapping = {'available_variables': self, 'searchpath': self.environment.loader}
        original = {}
        for key, value in kwargs.items():
            obj = mapping.get(key, self.environment)
            try:
                original[key] = getattr(obj, key)
                if value is not None:
                    setattr(obj, key, value)
            except AttributeError:
                pass
        yield
        for key in original:
            obj = mapping.get(key, self.environment)
            setattr(obj, key, original[key])

    def template(self, variable, convert_bare=False, preserve_trailing_newlines=True, escape_backslashes=True, fail_on_undefined=None, overrides=None, convert_data=True, static_vars=None, cache=None, disable_lookups=False):
        """
        Templates (possibly recursively) any given data as input. If convert_bare is
        set to True, the given data will be wrapped as a jinja2 variable ('{{foo}}')
        before being sent through the template engine.
        """
        static_vars = [] if static_vars is None else static_vars
        if cache is not None:
            display.deprecated('The `cache` option to `Templar.template` is no longer functional, and will be removed in a future release.', version='2.18')
        if hasattr(variable, '__UNSAFE__'):
            return variable
        if fail_on_undefined is None:
            fail_on_undefined = self._fail_on_undefined_errors
        if convert_bare:
            variable = self._convert_bare_variable(variable)
        if isinstance(variable, string_types):
            if not self.is_possibly_template(variable, overrides):
                return variable
            only_one = self.SINGLE_VAR.match(variable)
            if only_one:
                var_name = only_one.group(1)
                if var_name in self._available_variables:
                    resolved_val = self._available_variables[var_name]
                    if isinstance(resolved_val, NON_TEMPLATED_TYPES):
                        return resolved_val
                    elif resolved_val is None:
                        return C.DEFAULT_NULL_REPRESENTATION
            result = self.do_template(variable, preserve_trailing_newlines=preserve_trailing_newlines, escape_backslashes=escape_backslashes, fail_on_undefined=fail_on_undefined, overrides=overrides, disable_lookups=disable_lookups, convert_data=convert_data)
            self._compile_single_var(self.environment)
            return result
        elif is_sequence(variable):
            return [self.template(v, preserve_trailing_newlines=preserve_trailing_newlines, fail_on_undefined=fail_on_undefined, overrides=overrides, disable_lookups=disable_lookups) for v in variable]
        elif isinstance(variable, Mapping):
            d = {}
            for k in variable.keys():
                if k not in static_vars:
                    d[k] = self.template(variable[k], preserve_trailing_newlines=preserve_trailing_newlines, fail_on_undefined=fail_on_undefined, overrides=overrides, disable_lookups=disable_lookups)
                else:
                    d[k] = variable[k]
            return d
        else:
            return variable

    def is_template(self, data):
        """lets us know if data has a template"""
        if isinstance(data, string_types):
            return is_template(data, self.environment)
        elif isinstance(data, (list, tuple)):
            for v in data:
                if self.is_template(v):
                    return True
        elif isinstance(data, dict):
            for k in data:
                if self.is_template(k) or self.is_template(data[k]):
                    return True
        return False
    templatable = is_template

    def is_possibly_template(self, data, overrides=None):
        data, env = _create_overlay(data, overrides, self.environment)
        return is_possibly_template(data, env)

    def _convert_bare_variable(self, variable):
        """
        Wraps a bare string, which may have an attribute portion (ie. foo.bar)
        in jinja2 variable braces so that it is evaluated properly.
        """
        if isinstance(variable, string_types):
            contains_filters = '|' in variable
            first_part = variable.split('|')[0].split('.')[0].split('[')[0]
            if (contains_filters or first_part in self._available_variables) and self.environment.variable_start_string not in variable:
                return '%s%s%s' % (self.environment.variable_start_string, variable, self.environment.variable_end_string)
        return variable

    def _fail_lookup(self, name, *args, **kwargs):
        raise AnsibleError('The lookup `%s` was found, however lookups were disabled from templating' % name)

    def _now_datetime(self, utc=False, fmt=None):
        """jinja2 global function to return current datetime, potentially formatted via strftime"""
        if utc:
            now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        else:
            now = datetime.datetime.now()
        if fmt:
            return now.strftime(fmt)
        return now

    def _query_lookup(self, name, /, *args, **kwargs):
        """ wrapper for lookup, force wantlist true"""
        kwargs['wantlist'] = True
        return self._lookup(name, *args, **kwargs)

    def _lookup(self, name, /, *args, **kwargs):
        instance = lookup_loader.get(name, loader=self._loader, templar=self)
        if instance is None:
            raise AnsibleError('lookup plugin (%s) not found' % name)
        wantlist = kwargs.pop('wantlist', False)
        allow_unsafe = kwargs.pop('allow_unsafe', C.DEFAULT_ALLOW_UNSAFE_LOOKUPS)
        errors = kwargs.pop('errors', 'strict')
        loop_terms = listify_lookup_plugin_terms(terms=args, templar=self, fail_on_undefined=True, convert_bare=False)
        try:
            ran = instance.run(loop_terms, variables=self._available_variables, **kwargs)
        except (AnsibleUndefinedVariable, UndefinedError) as e:
            raise AnsibleUndefinedVariable(e)
        except AnsibleOptionsError as e:
            raise e
        except AnsibleLookupError as e:
            msg = 'Lookup failed but the error is being ignored: %s' % to_native(e)
            if errors == 'warn':
                display.warning(msg)
            elif errors == 'ignore':
                display.display(msg, log_only=True)
            else:
                raise e
            return [] if wantlist else None
        except Exception as e:
            msg = u"An unhandled exception occurred while running the lookup plugin '%s'. Error was a %s, original message: %s" % (name, type(e), to_text(e))
            if errors == 'warn':
                display.warning(msg)
            elif errors == 'ignore':
                display.display(msg, log_only=True)
            else:
                display.vvv('exception during Jinja2 execution: {0}'.format(format_exc()))
                raise AnsibleError(to_native(msg), orig_exc=e)
            return [] if wantlist else None
        if not is_sequence(ran):
            display.deprecated(f"The lookup plugin '{name}' was expected to return a list, got '{type(ran)}' instead. The lookup plugin '{name}' needs to be changed to return a list. This will be an error in Ansible 2.18", version='2.18')
        if ran and allow_unsafe is False:
            if self.cur_context:
                self.cur_context.unsafe = True
            if wantlist:
                return wrap_var(ran)
            try:
                if isinstance(ran[0], NativeJinjaText):
                    ran = wrap_var(NativeJinjaText(','.join(ran)))
                else:
                    ran = wrap_var(','.join(ran))
            except TypeError:
                if not isinstance(ran, Sequence):
                    raise AnsibleError("The lookup plugin '%s' did not return a list." % name)
                if len(ran) == 1:
                    ran = wrap_var(ran[0])
                else:
                    ran = wrap_var(ran)
            except KeyError:
                ran = wrap_var(','.join(ran))
        return ran

    def _make_undefined(self, hint=None):
        from jinja2.runtime import Undefined
        if hint is None or isinstance(hint, Undefined) or hint == '':
            hint = 'Mandatory variable has not been overridden'
        return AnsibleUndefined(hint)

    def do_template(self, data, preserve_trailing_newlines=True, escape_backslashes=True, fail_on_undefined=None, overrides=None, disable_lookups=False, convert_data=False):
        if self.jinja2_native and (not isinstance(data, string_types)):
            return data
        data_newlines = _count_newlines_from_end(data)
        if fail_on_undefined is None:
            fail_on_undefined = self._fail_on_undefined_errors
        try:
            data, myenv = _create_overlay(data, overrides, self.environment)
            self._compile_single_var(myenv)
            if escape_backslashes:
                data = _escape_backslashes(data, myenv)
            try:
                t = myenv.from_string(data)
            except (TemplateSyntaxError, SyntaxError) as e:
                raise AnsibleError('template error while templating string: %s. String: %s' % (to_native(e), to_native(data)), orig_exc=e)
            except Exception as e:
                if 'recursion' in to_native(e):
                    raise AnsibleError('recursive loop detected in template string: %s' % to_native(data), orig_exc=e)
                else:
                    return data
            if disable_lookups:
                t.globals['query'] = t.globals['q'] = t.globals['lookup'] = self._fail_lookup
            jvars = AnsibleJ2Vars(self, t.globals)
            cached_context = self.cur_context
            myenv.concat = myenv.__class__.concat
            if not self.jinja2_native and (not convert_data):
                myenv.concat = ansible_concat
            self.cur_context = t.new_context(jvars, shared=True)
            rf = t.root_render_func(self.cur_context)
            try:
                res = myenv.concat(rf)
                unsafe = getattr(self.cur_context, 'unsafe', False)
                if unsafe:
                    res = wrap_var(res)
            except TypeError as te:
                if 'AnsibleUndefined' in to_native(te):
                    errmsg = 'Unable to look up a name or access an attribute in template string (%s).\n' % to_native(data)
                    errmsg += "Make sure your variable name does not contain invalid characters like '-': %s" % to_native(te)
                    raise AnsibleUndefinedVariable(errmsg, orig_exc=te)
                else:
                    display.debug('failing because of a type error, template data is: %s' % to_text(data))
                    raise AnsibleError('Unexpected templating type error occurred on (%s): %s' % (to_native(data), to_native(te)), orig_exc=te)
            finally:
                self.cur_context = cached_context
            if isinstance(res, string_types) and preserve_trailing_newlines:
                res_newlines = _count_newlines_from_end(res)
                if data_newlines > res_newlines:
                    res += myenv.newline_sequence * (data_newlines - res_newlines)
                    if unsafe:
                        res = wrap_var(res)
            return res
        except (UndefinedError, AnsibleUndefinedVariable) as e:
            if fail_on_undefined:
                raise AnsibleUndefinedVariable(e, orig_exc=e)
            else:
                display.debug('Ignoring undefined failure: %s' % to_text(e))
                return data
    _do_template = do_template