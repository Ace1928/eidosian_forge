from typing import List
import pytest
import spacy
from spacy.training import Example
def _add_tagger_label(tagger, data):
    for _, annotations in data:
        for tag in annotations['tags']:
            tagger.add_label(tag)