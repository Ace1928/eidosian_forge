import os
import re
import time
from enum import Enum
from typing import List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader


        Args:
            contract_address: The address of the smart contract.
            blockchainType: The blockchain type.
            api_key: The Alchemy API key.
            startToken: The start token for pagination.
            get_all_tokens: Whether to get all tokens on the contract.
            max_execution_time: The maximum execution time (sec).
        