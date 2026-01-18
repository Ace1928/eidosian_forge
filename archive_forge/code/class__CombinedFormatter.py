import sys
import subprocess
import shlex
import os
import argparse
import shutil
import logging
import coloredlogs
class _CombinedFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass