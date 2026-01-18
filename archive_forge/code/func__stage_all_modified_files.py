import re
import ast
import logging
from typing import List, Dict, Any, Union
import numpy as np
import logging
from typing import List
import os
import logging
from typing import Dict, List, Union
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Union
import json
import xml.etree.ElementTree as ET
import logging
import os
import subprocess
import logging
from typing import List
import ast
import logging
from typing import List, Dict
import logging
from typing import Dict
import ast
import networkx as nx
import matplotlib.pyplot as plt
from typing import List
import logging
from typing import Type, Union
def _stage_all_modified_files(self) -> None:
    """
        Stage all modified files for commit using the version control system's command-line interface.
        """
    try:
        subprocess.run(['git', 'add', '.'], check=True)
    except subprocess.CalledProcessError as e:
        self.logger.error(f'Error occurred while staging modified files: {str(e)}')
        raise