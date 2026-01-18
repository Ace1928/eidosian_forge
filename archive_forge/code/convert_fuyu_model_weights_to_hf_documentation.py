import argparse
import os
import sys
import warnings
import flatdict
import torch
from transformers import FuyuConfig, FuyuForCausalLM, LlamaTokenizer
from transformers import FuyuForCausalLM, FuyuTokenizer

Sample usage: # TODO fix clone links from persimmon to fuyu
```
git clone https://github.com/adept-ai-labs/adept-inference
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_base_model_release.tar
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_chat_model_release.tar
python src/transformers/models/fuyu/convert_fuyu_weights_to_hf.py  --input_dir /path/to/downloaded/fuyu/weights/ --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import FuyuForCausalLM, FuyuTokenizer

model = FuyuForCausalLM.from_pretrained("/output/path")
tokenizer = FuyuTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
