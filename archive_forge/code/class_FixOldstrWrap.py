from __future__ import unicode_literals
import re
from lib2to3 import fixer_base
from lib2to3.pgen2 import token
from lib2to3.fixer_util import syms
from libfuturize.fixer_util import (future_import, touch_import_top,
class FixOldstrWrap(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = 'STRING'

    def transform(self, node, results):
        if node.type == token.STRING:
            touch_import_top(u'past.types', u'oldstr', node)
            if _literal_re.match(node.value):
                new = node.clone()
                new.prefix = u''
                new.value = u'b' + new.value
                wrapped = wrap_in_fn_call('oldstr', [new], prefix=node.prefix)
                return wrapped