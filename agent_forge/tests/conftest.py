import sys
from pathlib import Path

# add repo root to sys.path so 'core' can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
