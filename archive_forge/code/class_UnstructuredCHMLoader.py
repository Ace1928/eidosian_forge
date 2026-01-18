from typing import TYPE_CHECKING, Dict, List, Union
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
class UnstructuredCHMLoader(UnstructuredFileLoader):
    """Load `CHM` files using `Unstructured`.

    CHM means Microsoft Compiled HTML Help.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredCHMLoader

    loader = UnstructuredCHMLoader("example.chm")
    docs = loader.load()

    References
    ----------
    https://github.com/dottedmag/pychm
    http://www.jedrea.com/chmlib/
    """

    def _get_elements(self) -> List:
        from unstructured.partition.html import partition_html
        with CHMParser(self.file_path) as f:
            return [partition_html(text=item['content'], **self.unstructured_kwargs) for item in f.load_all()]