import re
import sys
import warnings
class IteratorWrapper:

    def __init__(self, wsgi_iterator, check_start_response):
        self.original_iterator = wsgi_iterator
        self.iterator = iter(wsgi_iterator)
        self.closed = False
        self.check_start_response = check_start_response

    def __iter__(self):
        return self

    def __next__(self):
        assert_(not self.closed, 'Iterator read after closed')
        v = next(self.iterator)
        if type(v) is not bytes:
            assert_(False, 'Iterator yielded non-bytestring (%r)' % (v,))
        if self.check_start_response is not None:
            assert_(self.check_start_response, 'The application returns and we started iterating over its body, but start_response has not yet been called')
            self.check_start_response = None
        return v

    def close(self):
        self.closed = True
        if hasattr(self.original_iterator, 'close'):
            self.original_iterator.close()

    def __del__(self):
        if not self.closed:
            sys.stderr.write('Iterator garbage collected without being closed')
        assert_(self.closed, 'Iterator garbage collected without being closed')