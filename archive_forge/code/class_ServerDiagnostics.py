from openstack import resource
class ServerDiagnostics(resource.Resource):
    resource_key = 'diagnostics'
    base_path = '/servers/%(server_id)s/diagnostics'
    allow_fetch = True
    requires_id = False
    _max_microversion = '2.48'
    has_config_drive = resource.Body('config_drive')
    state = resource.Body('state')
    driver = resource.Body('driver')
    hypervisor = resource.Body('hypervisor')
    hypervisor_os = resource.Body('hypervisor_os')
    uptime = resource.Body('uptime')
    num_cpus = resource.Body('num_cpus')
    num_disks = resource.Body('num_disks')
    num_nics = resource.Body('num_nics')
    memory_details = resource.Body('memory_details')
    cpu_details = resource.Body('cpu_details')
    disk_details = resource.Body('disk_details')
    nic_details = resource.Body('nic_details')
    server_id = resource.URI('server_id')