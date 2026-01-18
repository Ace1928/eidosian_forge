import html
from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...utils import is_bs4_available, logging, requires_backends

        Main method to prepare for the model one or several HTML strings.

        Args:
            html_strings (`str`, `List[str]`):
                The HTML string or batch of HTML strings from which to extract nodes and corresponding xpaths.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **nodes** -- Nodes.
            - **xpaths** -- Corresponding xpaths.

        Examples:

        ```python
        >>> from transformers import MarkupLMFeatureExtractor

        >>> page_name_1 = "page1.html"
        >>> page_name_2 = "page2.html"
        >>> page_name_3 = "page3.html"

        >>> with open(page_name_1) as f:
        ...     single_html_string = f.read()

        >>> feature_extractor = MarkupLMFeatureExtractor()

        >>> # single example
        >>> encoding = feature_extractor(single_html_string)
        >>> print(encoding.keys())
        >>> # dict_keys(['nodes', 'xpaths'])

        >>> # batched example

        >>> multi_html_strings = []

        >>> with open(page_name_2) as f:
        ...     multi_html_strings.append(f.read())
        >>> with open(page_name_3) as f:
        ...     multi_html_strings.append(f.read())

        >>> encoding = feature_extractor(multi_html_strings)
        >>> print(encoding.keys())
        >>> # dict_keys(['nodes', 'xpaths'])
        ```