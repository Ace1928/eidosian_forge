import json
import logging
import os
import re
import tempfile
import time
from abc import ABC
from io import StringIO
from pathlib import Path
from typing import (
from urllib.parse import urlparse
import requests
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.pdf import (
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
class AmazonTextractPDFLoader(BasePDFLoader):
    """Load `PDF` files from a local file system, HTTP or S3.

    To authenticate, the AWS client uses the following methods to
    automatically load credentials:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If a specific credential profile should be used, you must pass
    the name of the profile from the ~/.aws/credentials file that is to be used.

    Make sure the credentials / roles used have the required policies to
    access the Amazon Textract service.

    Example:
        .. code-block:: python
            from langchain_community.document_loaders import AmazonTextractPDFLoader
            loader = AmazonTextractPDFLoader(
                file_path="s3://pdfs/myfile.pdf"
            )
            document = loader.load()
    """

    def __init__(self, file_path: str, textract_features: Optional[Sequence[str]]=None, client: Optional[Any]=None, credentials_profile_name: Optional[str]=None, region_name: Optional[str]=None, endpoint_url: Optional[str]=None, headers: Optional[Dict]=None, *, linearization_config: Optional['TextLinearizationConfig']=None) -> None:
        """Initialize the loader.

        Args:
            file_path: A file, url or s3 path for input file
            textract_features: Features to be used for extraction, each feature
                               should be passed as a str that conforms to the enum
                               `Textract_Features`, see `amazon-textract-caller` pkg
            client: boto3 textract client (Optional)
            credentials_profile_name: AWS profile name, if not default (Optional)
            region_name: AWS region, eg us-east-1 (Optional)
            endpoint_url: endpoint url for the textract service (Optional)
            linearization_config: Config to be used for linearization of the output
                                  should be an instance of TextLinearizationConfig from
                                  the `textractor` pkg
        """
        super().__init__(file_path, headers=headers)
        try:
            import textractcaller as tc
        except ImportError:
            raise ModuleNotFoundError('Could not import amazon-textract-caller python package. Please install it with `pip install amazon-textract-caller`.')
        if textract_features:
            features = [tc.Textract_Features[x] for x in textract_features]
        else:
            features = []
        if credentials_profile_name or region_name or endpoint_url:
            try:
                import boto3
                if credentials_profile_name is not None:
                    session = boto3.Session(profile_name=credentials_profile_name)
                else:
                    session = boto3.Session()
                client_params = {}
                if region_name:
                    client_params['region_name'] = region_name
                if endpoint_url:
                    client_params['endpoint_url'] = endpoint_url
                client = session.client('textract', **client_params)
            except ImportError:
                raise ModuleNotFoundError('Could not import boto3 python package. Please install it with `pip install boto3`.')
            except Exception as e:
                raise ValueError(f'Could not load credentials to authenticate with AWS client. Please check that credentials in the specified profile name are valid. {e}') from e
        self.parser = AmazonTextractPDFParser(textract_features=features, client=client, linearization_config=linearization_config)

    def load(self) -> List[Document]:
        """Load given path as pages."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents"""
        if self.web_path and self._is_s3_url(self.web_path):
            blob = Blob(path=self.web_path)
        else:
            blob = Blob.from_path(self.file_path)
            if AmazonTextractPDFLoader._get_number_of_pages(blob) > 1:
                raise ValueError(f'the file {blob.path} is a multi-page document,                     but not stored on S3.                     Textract requires multi-page documents to be on S3.')
        yield from self.parser.parse(blob)

    @staticmethod
    def _get_number_of_pages(blob: Blob) -> int:
        try:
            import pypdf
            from PIL import Image, ImageSequence
        except ImportError:
            raise ModuleNotFoundError('Could not import pypdf or Pilloe python package. Please install it with `pip install pypdf Pillow`.')
        if blob.mimetype == 'application/pdf':
            with blob.as_bytes_io() as input_pdf_file:
                pdf_reader = pypdf.PdfReader(input_pdf_file)
                return len(pdf_reader.pages)
        elif blob.mimetype == 'image/tiff':
            num_pages = 0
            img = Image.open(blob.as_bytes())
            for _, _ in enumerate(ImageSequence.Iterator(img)):
                num_pages += 1
            return num_pages
        elif blob.mimetype in ['image/png', 'image/jpeg']:
            return 1
        else:
            raise ValueError(f'unsupported mime type: {blob.mimetype}')