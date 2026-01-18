import argparse
from transformers import T5Config, T5ForConditionalGeneration, load_tf_weights_in_t5
from transformers.utils import logging
Convert T5 checkpoint.