from typing import Dict, List, Optional, Tuple, Union
from tokenizers import AddedToken, EncodeInput, Encoding, InputSequence, Tokenizer
from tokenizers.decoders import Decoder
from tokenizers.models import Model
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import PostProcessor
Apply all the post-processing steps to the given encodings.

        The various steps are:
            1. Truncate according to global params (provided to `enable_truncation`)
            2. Apply the PostProcessor
            3. Pad according to global params. (provided to `enable_padding`)

        Args:
            encoding: Encoding:
                The main Encoding to post process

            pair: Optional[Encoding]:
                An optional pair Encoding

            add_special_tokens: bool:
                Whether to add special tokens

        Returns:
            The resulting Encoding
        