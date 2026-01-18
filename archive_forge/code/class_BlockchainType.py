import os
import re
import time
from enum import Enum
from typing import List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
class BlockchainType(Enum):
    """Enumerator of the supported blockchains."""
    ETH_MAINNET = 'eth-mainnet'
    ETH_GOERLI = 'eth-goerli'
    POLYGON_MAINNET = 'polygon-mainnet'
    POLYGON_MUMBAI = 'polygon-mumbai'