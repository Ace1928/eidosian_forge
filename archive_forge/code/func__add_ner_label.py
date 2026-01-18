from typing import List
import pytest
import spacy
from spacy.training import Example
def _add_ner_label(ner, data):
    for _, annotations in data:
        for ent in annotations['entities']:
            ner.add_label(ent[2])