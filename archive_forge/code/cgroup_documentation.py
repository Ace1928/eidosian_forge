from __future__ import annotations
import codecs
import dataclasses
import pathlib
import re
Parse the given output from the proc filesystem and return a tuple of mount info entries.