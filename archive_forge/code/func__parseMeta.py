from __future__ import annotations
import json
import logging
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _parseMeta(self, nvmap: dict, filecol: str) -> Tuple[List[str], List[str], str]:
    filepath = ''
    if filecol == '':
        nvec = list(nvmap.keys())
        vvec = list(nvmap.values())
    else:
        nvec = []
        vvec = []
        if filecol in nvmap:
            nvec.append(filecol)
            vvec.append(nvmap[filecol])
            filepath = nvmap[filecol]
        for k, v in nvmap.items():
            if k != filecol:
                nvec.append(k)
                vvec.append(v)
    vvec_s = [str(e) for e in vvec]
    return (nvec, vvec_s, filepath)