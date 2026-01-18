import threading
from tensorboard import errors
def _opCreationDigestToDataObject(self, op_creation_digest, graph):
    if op_creation_digest is None:
        return None
    json_object = op_creation_digest.to_json()
    del json_object['graph_id']
    json_object['graph_ids'] = self._getGraphStackIds(op_creation_digest.graph_id)
    json_object['num_outputs'] = op_creation_digest.num_outputs
    del json_object['input_names']
    json_object['inputs'] = []
    for input_tensor_name in op_creation_digest.input_names or []:
        input_op_name, output_slot = parse_tensor_name(input_tensor_name)
        json_object['inputs'].append({'op_name': input_op_name, 'output_slot': output_slot})
    json_object['consumers'] = []
    for _ in range(json_object['num_outputs']):
        json_object['consumers'].append([])
    for src_slot, consumer_op_name, dst_slot in graph.get_op_consumers(json_object['op_name']):
        json_object['consumers'][src_slot].append({'op_name': consumer_op_name, 'input_slot': dst_slot})
    return json_object