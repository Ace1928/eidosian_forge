import inspect
import io
import logging
import re
import sys
import textwrap
from pyomo.version.info import releaselevel
from pyomo.common.deprecation import deprecated
from pyomo.common.fileutils import PYOMO_ROOT_DIR
from pyomo.common.formatting import wrap_reStructuredText
class WrappingFormatter(logging.Formatter):
    _flag = '<<!MSG!>>'

    def __init__(self, **kwds):
        if 'fmt' not in kwds:
            if kwds.get('style', '%') == '%':
                kwds['fmt'] = '%(levelname)s: %(message)s'
            elif kwds['style'] == '{':
                kwds['fmt'] = '{levelname}: {message}'
            elif kwds['style'] == '$':
                kwds['fmt'] = '$levelname: $message'
            else:
                raise ValueError('unrecognized style flag "%s"' % (kwds['style'],))
        self._wrapper = textwrap.TextWrapper(width=kwds.pop('wrap', 78))
        self._wrapper.subsequent_indent = kwds.pop('hang', ' ' * 4)
        if not self._wrapper.subsequent_indent:
            self._wrapper.subsequent_indent = ''
        self.basepath = kwds.pop('base', None)
        super(WrappingFormatter, self).__init__(**kwds)

    def format(self, record):
        msg = record.getMessage()
        if record.msg.__class__ is not str and isinstance(record.msg, Preformatted):
            return msg
        _orig = {k: getattr(record, k) for k in ('msg', 'args', 'pathname', 'levelname')}
        _id = getattr(record, 'id', None)
        record.msg = self._flag
        record.args = None
        if _id:
            record.levelname += f' ({_id.upper()})'
        if self.basepath and record.pathname.startswith(self.basepath):
            record.pathname = '[base]' + record.pathname[len(self.basepath):]
        try:
            raw_msg = super(WrappingFormatter, self).format(record)
        finally:
            for k, v in _orig.items():
                setattr(record, k, v)
        if getattr(record, 'cleandoc', True):
            msg = inspect.cleandoc(msg)
        return '\n'.join((self._wrap_msg(line, msg, _id) if self._flag in line else line for line in raw_msg.splitlines()))

    def _wrap_msg(self, format_line, msg, _id):
        _init = (self._wrapper.initial_indent, self._wrapper.subsequent_indent)
        indent = _indentation_re.match(format_line).group()
        if indent:
            self._wrapper.initial_indent = self._wrapper.subsequent_indent = indent
        try:
            wrapped_msg = wrap_reStructuredText(format_line.strip().replace(self._flag, msg), self._wrapper)
        finally:
            self._wrapper.initial_indent, self._wrapper.subsequent_indent = _init
        if _id:
            wrapped_msg += f'\n{indent}{_init[1]}See also {RTD(_id)}'
        return wrapped_msg