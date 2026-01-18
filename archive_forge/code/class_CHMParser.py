from typing import TYPE_CHECKING, Dict, List, Union
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
class CHMParser(object):
    """Microsoft Compiled HTML Help (CHM) Parser."""
    path: str
    file: 'chm.CHMFile'

    def __init__(self, path: str):
        from chm import chm
        self.path = path
        self.file = chm.CHMFile()
        self.file.LoadCHM(path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.CloseCHM()

    @property
    def encoding(self) -> str:
        return self.file.GetEncoding().decode('utf-8')

    def index(self) -> List[Dict[str, str]]:
        from urllib.parse import urlparse
        from bs4 import BeautifulSoup
        res = []
        index = self.file.GetTopicsTree().decode(self.encoding)
        soup = BeautifulSoup(index)
        for obj in soup.find_all('object'):
            name = ''
            local = ''
            for param in obj.find_all('param'):
                if param['name'] == 'Name':
                    name = param['value']
                if param['name'] == 'Local':
                    local = param['value']
            if not name or not local:
                continue
            local = urlparse(local).path
            if not local.startswith('/'):
                local = '/' + local
            res.append({'name': name, 'local': local})
        return res

    def load(self, path: Union[str, bytes]) -> str:
        if isinstance(path, str):
            path = path.encode('utf-8')
        obj = self.file.ResolveObject(path)[1]
        return self.file.RetrieveObject(obj)[1].decode(self.encoding)

    def load_all(self) -> List[Dict[str, str]]:
        res = []
        index = self.index()
        for item in index:
            content = self.load(item['local'])
            res.append({'name': item['name'], 'local': item['local'], 'content': content})
        return res