from datetime import datetime
from boto.compat import six
class EnvironmentResourceDescription(BaseObject):

    def __init__(self, response):
        super(EnvironmentResourceDescription, self).__init__()
        self.auto_scaling_groups = []
        if response['AutoScalingGroups']:
            for member in response['AutoScalingGroups']:
                auto_scaling_group = AutoScalingGroup(member)
                self.auto_scaling_groups.append(auto_scaling_group)
        self.environment_name = str(response['EnvironmentName'])
        self.instances = []
        if response['Instances']:
            for member in response['Instances']:
                instance = Instance(member)
                self.instances.append(instance)
        self.launch_configurations = []
        if response['LaunchConfigurations']:
            for member in response['LaunchConfigurations']:
                launch_configuration = LaunchConfiguration(member)
                self.launch_configurations.append(launch_configuration)
        self.load_balancers = []
        if response['LoadBalancers']:
            for member in response['LoadBalancers']:
                load_balancer = LoadBalancer(member)
                self.load_balancers.append(load_balancer)
        self.triggers = []
        if response['Triggers']:
            for member in response['Triggers']:
                trigger = Trigger(member)
                self.triggers.append(trigger)