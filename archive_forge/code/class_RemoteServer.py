import jinja2
from minerl.herobraine.hero.handler import Handler
class RemoteServer(Handler):

    def __init__(self, address: str):
        self.address = address

    def xml_template(self) -> str:
        address = self.address
        if callable(address):
            address = address()
        if not isinstance(address, str):
            raise ValueError(f'address should be a string (provided {address})')
        return f'<RemoteServer>{address}</RemoteServer>'

    def to_string(self) -> str:
        return 'remote_server'