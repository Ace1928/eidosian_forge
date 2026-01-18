import datetime
from redis.utils import str_if_bytes
def _parse_node_line(line):
    line_items = line.split(' ')
    node_id, addr, flags, master_id, ping, pong, epoch, connected = line.split(' ')[:8]
    addr = addr.split('@')[0]
    node_dict = {'node_id': node_id, 'flags': flags, 'master_id': master_id, 'last_ping_sent': ping, 'last_pong_rcvd': pong, 'epoch': epoch, 'slots': [], 'migrations': [], 'connected': True if connected == 'connected' else False}
    if len(line_items) >= 9:
        slots, migrations = _parse_slots(line_items[8:])
        node_dict['slots'], node_dict['migrations'] = (slots, migrations)
    return (addr, node_dict)