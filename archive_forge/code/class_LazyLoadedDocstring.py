from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.method import document_model_driven_method
from botocore.docs.paginator import document_paginate_method
from botocore.docs.waiter import document_wait_method
class LazyLoadedDocstring(str):
    """Used for lazily loading docstrings

    You can instantiate this class and assign it to a __doc__ value.
    The docstring will not be generated till accessed via __doc__ or
    help(). Note that all docstring classes **must** subclass from
    this class. It cannot be used directly as a docstring.
    """

    def __init__(self, *args, **kwargs):
        """
        The args and kwargs are the same as the underlying document
        generation function. These just get proxied to the underlying
        function.
        """
        super().__init__()
        self._gen_args = args
        self._gen_kwargs = kwargs
        self._docstring = None

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def _write_docstring(self, *args, **kwargs):
        raise NotImplementedError('_write_docstring is not implemented. Please subclass from this class and provide your own _write_docstring method')

    def expandtabs(self, tabsize=8):
        """Expands tabs to spaces

        So this is a big hack in order to get lazy loaded docstring work
        for the ``help()``. In the ``help()`` function, ``pydoc`` and
        ``inspect`` are used. At some point the ``inspect.cleandoc``
        method is called. To clean the docs ``expandtabs`` is called
        and that is where we override the method to generate and return the
        docstrings.
        """
        if self._docstring is None:
            self._generate()
        return self._docstring.expandtabs(tabsize)

    def __str__(self):
        return self._generate()
    __repr__ = __str__

    def _generate(self):
        if self._docstring is None:
            self._docstring = self._create_docstring()
        return self._docstring

    def _create_docstring(self):
        docstring_structure = DocumentStructure('docstring', target='html')
        self._write_docstring(docstring_structure, *self._gen_args, **self._gen_kwargs)
        return docstring_structure.flush_structure().decode('utf-8')