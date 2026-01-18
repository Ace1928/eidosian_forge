from minerl.herobraine.hero.handler import Handler
import jinja2
class SpawningInitialCondition(Handler):

    def to_string(self) -> str:
        return 'spawning_initial_condition'

    def xml_template(self) -> str:
        return str('<AllowSpawning>{{allow_spawning | string | lower}}</AllowSpawning>')

    def __init__(self, allow_spawning: bool):
        self.allow_spawning = allow_spawning