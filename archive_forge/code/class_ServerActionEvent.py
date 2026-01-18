from openstack import resource
class ServerActionEvent(resource.Resource):
    _max_microversion = '2.84'
    event = resource.Body('event')
    start_time = resource.Body('start_time')
    finish_time = resource.Body('finish_time')
    result = resource.Body('result')
    traceback = resource.Body('traceback')
    host = resource.Body('host')
    host_id = resource.Body('hostId')
    details = resource.Body('details')