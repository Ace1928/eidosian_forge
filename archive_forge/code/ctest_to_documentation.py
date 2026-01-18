import argparse
import collections
import io
import json
import logging
import os
import sys
from xml.etree import ElementTree as ET
from cmakelang.format import __main__
from cmakelang import lex
from cmakelang import parse
Return an XML representation of the test specification.