import json
from argparse import ArgumentParser
from pathlib import Path

This script converts Falcon custom code checkpoints to modern Falcon checkpoints that use code in the Transformers
library. After conversion, performance (especially for generation) should improve and the checkpoint can be loaded
without needing trust_remote_code=True.
