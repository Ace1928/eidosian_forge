import argparse
from t5x import checkpoints
from transformers import FlaxT5ForConditionalGeneration, T5Config
Convert T5X checkpoints from the original repository to JAX/FLAX model.