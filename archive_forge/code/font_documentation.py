from __future__ import annotations
import os
import shutil
from typing import (
import fs.base
import fs.tempfs
from attrs import define, field
from fontTools.ufoLib import UFOFileStructure, UFOReader, UFOWriter
from ufoLib2.constants import DEFAULT_LAYER_NAME
from ufoLib2.objects.dataSet import DataSet
from ufoLib2.objects.features import Features
from ufoLib2.objects.glyph import Glyph
from ufoLib2.objects.guideline import Guideline
from ufoLib2.objects.imageSet import ImageSet
from ufoLib2.objects.info import Info
from ufoLib2.objects.kerning import Kerning, KerningPair
from ufoLib2.objects.layer import Layer
from ufoLib2.objects.layerSet import LayerSet
from ufoLib2.objects.lib import (
from ufoLib2.objects.misc import (
from ufoLib2.serde import serde
from ufoLib2.typing import HasIdentifier, PathLike, T
Saves the font to ``path``.

        Args:
            path: The target path. If it is None, the path from the last save (except
                when that was a ``fs.base.FS``) or when the font was first opened will
                be used.
            formatVersion: The version to save the UFO as. Only version 3 is supported
                currently.
            structure (fontTools.ufoLib.UFOFileStructure): How to store the UFO.
                Can be either None, "zip" or "package". If None, it tries to use the
                same structure as the original UFO at the output path. If "zip", the
                UFO will be saved as compressed archive. If "package", it is saved as
                a regular folder or "package".
            overwrite: If False, raises OSError when the target path exists. If True,
                overwrites the target path.
            validate: If True, will validate the data in Font before writing it out. If
                False, will write out whatever is serializable.
        