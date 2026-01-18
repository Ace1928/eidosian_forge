from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, JINJA2_SUSPEND
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from pydevd_file_utils import canonical_normalized_path
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode
from _pydev_bundle import pydev_log
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle.pydevd_api import PyDevdAPI
class Jinja2TemplateFrame(object):
    IS_PLUGIN_FRAME = True

    def __init__(self, frame, original_filename=None, template_lineno=None):
        if original_filename is None:
            original_filename = _get_jinja2_template_original_filename(frame)
        if template_lineno is None:
            template_lineno = _get_jinja2_template_line(frame)
        self.back_context = None
        if 'context' in frame.f_locals:
            self.back_context = frame.f_locals['context']
        self.f_code = FCode('template', original_filename)
        self.f_lineno = template_lineno
        self.f_back = frame
        self.f_globals = {}
        self.f_locals = self.collect_context(frame)
        self.f_trace = None

    def _get_real_var_name(self, orig_name):
        parts = orig_name.split('_')
        if len(parts) > 1 and parts[0].isdigit():
            return parts[1]
        return orig_name

    def collect_context(self, frame):
        res = {}
        for k, v in frame.f_locals.items():
            if not k.startswith('l_'):
                res[k] = v
            elif v and (not _is_missing(v)):
                res[self._get_real_var_name(k[2:])] = v
        if self.back_context is not None:
            for k, v in self.back_context.items():
                res[k] = v
        return res

    def _change_variable(self, frame, name, value):
        in_vars_or_parents = False
        if 'context' in frame.f_locals:
            if name in frame.f_locals['context'].parent:
                self.back_context.parent[name] = value
                in_vars_or_parents = True
            if name in frame.f_locals['context'].vars:
                self.back_context.vars[name] = value
                in_vars_or_parents = True
        l_name = 'l_' + name
        if l_name in frame.f_locals:
            if in_vars_or_parents:
                frame.f_locals[l_name] = self.back_context.resolve(name)
            else:
                frame.f_locals[l_name] = value