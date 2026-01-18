from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.parsers.generic import MimeTypeBasedParser
from langchain_community.document_loaders.parsers.msword import MsWordParser
from langchain_community.document_loaders.parsers.pdf import PyMuPDFParser
from langchain_community.document_loaders.parsers.txt import TextParser
def _get_default_parser() -> BaseBlobParser:
    """Get default mime-type based parser."""
    return MimeTypeBasedParser(handlers={'application/pdf': PyMuPDFParser(), 'text/plain': TextParser(), 'application/msword': MsWordParser(), 'application/vnd.openxmlformats-officedocument.wordprocessingml.document': MsWordParser()}, fallback_parser=None)