from abc import abstractmethod
import types
from minerl.herobraine.hero.handlers.translation import TranslationHandler
import typing
from minerl.herobraine.hero.spaces import Dict
from minerl.herobraine.hero.handler import Handler
from typing import List
import jinja2
import gym
from lxml import etree
import os
import abc
import importlib
from minerl.herobraine.hero import spaces
Consolidates duplicate XML representations from the handlers.

        Deduplication happens by first getting all of the handler.xml() strings
        of the handlers, and then converting them into etrees. After that we check
        if the there are any top level elements that are duplicated and pick the first of them
        to retain. We then convert the remaining etrees back into strings and join them with new lines.

        Args:
            handlers (List[Handler]): A list of handlers to consolidate.

        Returns:
            str: The XML
        