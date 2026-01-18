import jinja2
from minerl.herobraine.hero.handler import Handler
class FlatWorldGenerator(Handler):
    """Generates a world that is a flat landscape."""

    def to_string(self) -> str:
        return 'flat_world_generator'

    def xml_template(self) -> str:
        return str('<FlatWorldGenerator\n                forceReset="{{force_reset | string | lower}}"\n                generatorString="{{generatorString}}" />\n            ')

    def __init__(self, force_reset: bool=True, generatorString: str=''):
        self.force_reset = force_reset
        self.generatorString = generatorString