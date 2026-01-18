import sqlite3
import os
from tqdm import tqdm
from collections import deque
import random
from parlai.core.teachers import create_task_agent_from_taskname
import parlai.utils.logging as logging

    Preprocess and store a corpus of documents in sqlite.

    Args:
        task: ParlAI tasks of text (and possibly values) to store.
        save_path: Path to output sqlite db.
        num_workers: Number of parallel processes to use when reading docs.
    