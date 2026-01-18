import jinja2
from minerl.herobraine.hero.handler import Handler
class BiomeGenerator(Handler):

    def to_string(self) -> str:
        return 'biome_generator'

    def xml_template(self) -> str:
        return str('<BiomeGenerator\n                forceReset="{{force_reset | string | lower}}"\n                biome="{{biome_id}}" />\n            ')

    def __init__(self, biome_id: int, force_reset: bool=True):
        self.biome_id = biome_id
        self.force_reset = force_reset