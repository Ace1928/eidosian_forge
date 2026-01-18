import logging
from oslo_utils import timeutils
from suds import sudsobject
class WithRetrieval(object):
    """Context to retrieve results.

    This context provides an iterator to retrieve results and cancel (when
    needed) retrieve operation on __exit__.

    Example:

      with WithRetrieval(vim, retrieve_result) as objects:
          for obj in objects:
              # Use obj
    """

    def __init__(self, vim, retrieve_result):
        super(WithRetrieval, self).__init__()
        self.vim = vim
        self.retrieve_result = retrieve_result

    def __enter__(self):
        return iter(self)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.retrieve_result:
            cancel_retrieval(self.vim, self.retrieve_result)

    def __iter__(self):
        while self.retrieve_result:
            for obj in self.retrieve_result.objects:
                yield obj
            self.retrieve_result = continue_retrieval(self.vim, self.retrieve_result)