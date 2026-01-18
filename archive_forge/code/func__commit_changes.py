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
def _commit_changes(self, commit_message: str) -> None:
    """
        Commit the staged changes with the provided commit message.

        Args:
            commit_message (str): The commit message describing the changes.
        """
    try:
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
    except subprocess.CalledProcessError as e:
        self.logger.error(f'Error occurred while committing changes: {str(e)}')
        raise