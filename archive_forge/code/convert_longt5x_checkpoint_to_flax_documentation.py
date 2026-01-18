import argparse
from t5x import checkpoints
from transformers import AutoConfig, FlaxAutoModelForSeq2SeqLM
Convert T5/LongT5X checkpoints from the original repository to JAX/FLAX model. This script is an extension of
'src/transformers/models/t5/convert_t5x_checkpoint_to_flax.
