import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='string')
class _StringChecks(SyntaxRule):
    message = 'bytes can only contain ASCII literal characters.'

    def is_issue(self, leaf):
        string_prefix = leaf.string_prefix.lower()
        if 'b' in string_prefix and any((c for c in leaf.value if ord(c) > 127)):
            return True
        if 'r' not in string_prefix:
            payload = leaf._get_payload()
            if 'b' in string_prefix:
                payload = payload.encode('utf-8')
                func = codecs.escape_decode
            else:
                func = codecs.unicode_escape_decode
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    func(payload)
            except UnicodeDecodeError as e:
                self.add_issue(leaf, message='(unicode error) ' + str(e))
            except ValueError as e:
                self.add_issue(leaf, message='(value error) ' + str(e))