import jinja2
from minerl.herobraine.hero.handler import Handler
class FileWorldGenerator(Handler):
    """Generates a world from a file."""

    def to_string(self) -> str:
        return 'file_world_generator'

    def xml_template(self) -> str:
        return str('<FileWorldGenerator\n                destroyAfterUse = "{{destroy_after_use | string | lower}}"\n                src = "{{filename}}" />\n            ')

    def __init__(self, filename: str, destroy_after_use: bool=True):
        self.filename = filename
        self.destroy_after_use = destroy_after_use