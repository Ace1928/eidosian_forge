import argparse
import gc
import json
import os
import shutil
from pathlib import Path
import torch
import yaml
from tokenizers import Tokenizer
from transformers import OlmoConfig, OlmoForCausalLM
from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast
from transformers import OlmoForCausalLM, AutoTokenizer

Sample usage:

```
python src/transformers/models/olmo/convert_olmo_weights_to_hf.py     --input_dir /path/to/downloaded/olmo/weights --model_size 7B --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import OlmoForCausalLM, AutoTokenizer

model = OlmoForCausalLM.from_pretrained("/output/path")
tokenizer = AutoTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
