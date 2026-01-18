import argparse
import os
import warnings
import torch
from accelerate import init_empty_weights
from transformers import GemmaTokenizer, RecurrentGemmaConfig, RecurrentGemmaForCausalLM
import regex as re
from transformers import GemmaForCausalLM, GemmaTokenizerFast

Sample usage:

```
python src/transformers/models/gemma/convert_gemma_weights_to_hf.py     --input_dir /path/to/downloaded/gemma/weights --model_size 7B --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import GemmaForCausalLM, GemmaTokenizerFast

model = GemmaForCausalLM.from_pretrained("/output/path")
tokenizer = GemmaTokenizerFast.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
