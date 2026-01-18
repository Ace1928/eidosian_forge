from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class PreferredSpawnBiome(Handler):

    def __init__(self, biome):
        self.biome = biome

    def to_string(self):
        return 'preferred_spawn_biome'

    def xml_template(self) -> str:
        _biome = None
        if callable(self.biome):
            _biome = self.biome()
        elif isinstance(self.biome, str):
            _biome = self.biome
        if _biome is not None:
            return f'<PreferredSpawnBiome>{_biome}</PreferredSpawnBiome>'
        return ''