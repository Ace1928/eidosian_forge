from __future__ import annotations
import warnings
from typing import (
from urllib.parse import urlparse
import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
class AmazonTextractPDFParser(BaseBlobParser):
    """Send `PDF` files to `Amazon Textract` and parse them.

    For parsing multi-page PDFs, they have to reside on S3.

    The AmazonTextractPDFLoader calls the
    [Amazon Textract Service](https://aws.amazon.com/textract/)
    to convert PDFs into a Document structure.
    Single and multi-page documents are supported with up to 3000 pages
    and 512 MB of size.

    For the call to be successful an AWS account is required,
    similar to the
    [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)
    requirements.

    Besides the AWS configuration, it is very similar to the other PDF
    loaders, while also supporting JPEG, PNG and TIFF and non-native
    PDF formats.

    ```python
    from langchain_community.document_loaders import AmazonTextractPDFLoader
    loader=AmazonTextractPDFLoader("example_data/alejandro_rosalez_sample-small.jpeg")
    documents = loader.load()
    ```

    One feature is the linearization of the output.
    When using the features LAYOUT, FORMS or TABLES together with Textract

    ```python
    from langchain_community.document_loaders import AmazonTextractPDFLoader
    # you can mix and match each of the features
    loader=AmazonTextractPDFLoader(
        "example_data/alejandro_rosalez_sample-small.jpeg",
        textract_features=["TABLES", "LAYOUT"])
    documents = loader.load()
    ```

    it will generate output that formats the text in reading order and
    try to output the information in a tabular structure or
    output the key/value pairs with a colon (key: value).
    This helps most LLMs to achieve better accuracy when
    processing these texts.

    """

    def __init__(self, textract_features: Optional[Sequence[int]]=None, client: Optional[Any]=None, *, linearization_config: Optional['TextLinearizationConfig']=None) -> None:
        """Initializes the parser.

        Args:
            textract_features: Features to be used for extraction, each feature
                               should be passed as an int that conforms to the enum
                               `Textract_Features`, see `amazon-textract-caller` pkg
            client: boto3 textract client
            linearization_config: Config to be used for linearization of the output
                                  should be an instance of TextLinearizationConfig from
                                  the `textractor` pkg
        """
        try:
            import textractcaller as tc
            import textractor.entities.document as textractor
            self.tc = tc
            self.textractor = textractor
            if textract_features is not None:
                self.textract_features = [tc.Textract_Features(f) for f in textract_features]
            else:
                self.textract_features = []
            if linearization_config is not None:
                self.linearization_config = linearization_config
            else:
                self.linearization_config = self.textractor.TextLinearizationConfig(hide_figure_layout=True, title_prefix='# ', section_header_prefix='## ', list_element_prefix='*')
        except ImportError:
            raise ImportError('Could not import amazon-textract-caller or amazon-textract-textractor python package. Please install it with `pip install amazon-textract-caller` & `pip install amazon-textract-textractor`.')
        if not client:
            try:
                import boto3
                self.boto3_textract_client = boto3.client('textract')
            except ImportError:
                raise ImportError('Could not import boto3 python package. Please install it with `pip install boto3`.')
        else:
            self.boto3_textract_client = client

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Iterates over the Blob pages and returns an Iterator with a Document
        for each page, like the other parsers If multi-page document, blob.path
        has to be set to the S3 URI and for single page docs
        the blob.data is taken
        """
        url_parse_result = urlparse(str(blob.path)) if blob.path else None
        if url_parse_result and url_parse_result.scheme == 's3' and url_parse_result.netloc:
            textract_response_json = self.tc.call_textract(input_document=str(blob.path), features=self.textract_features, boto3_textract_client=self.boto3_textract_client)
        else:
            textract_response_json = self.tc.call_textract(input_document=blob.as_bytes(), features=self.textract_features, call_mode=self.tc.Textract_Call_Mode.FORCE_SYNC, boto3_textract_client=self.boto3_textract_client)
        document = self.textractor.Document.open(textract_response_json)
        for idx, page in enumerate(document.pages):
            yield Document(page_content=page.get_text(config=self.linearization_config), metadata={'source': blob.source, 'page': idx + 1})