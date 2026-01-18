import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
class TestsWithEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_id = str(random.randint(1, 1000000))
        cls.app_name = 'app-' + cls.random_id
        cls.environment = 'environment-' + cls.random_id
        cls.template = 'template-' + cls.random_id
        cls.beanstalk = Layer1Wrapper()
        cls.beanstalk.create_application(application_name=cls.app_name)
        cls.beanstalk.create_configuration_template(application_name=cls.app_name, template_name=cls.template, solution_stack_name='32bit Amazon Linux running Tomcat 6')
        cls.app_version = 'version-' + cls.random_id
        cls.beanstalk.create_application_version(application_name=cls.app_name, version_label=cls.app_version)
        cls.beanstalk.create_environment(cls.app_name, cls.environment, template_name=cls.template)
        cls.wait_for_env(cls.environment)

    @classmethod
    def tearDownClass(cls):
        cls.beanstalk.delete_application(application_name=cls.app_name, terminate_env_by_force=True)
        cls.wait_for_env(cls.environment, 'Terminated')

    @classmethod
    def wait_for_env(cls, env_name, status='Ready'):
        while not cls.env_ready(env_name, status):
            time.sleep(15)

    @classmethod
    def env_ready(cls, env_name, desired_status):
        result = cls.beanstalk.describe_environments(application_name=cls.app_name, environment_names=env_name)
        status = result.environments[0].status
        return status == desired_status

    def test_describe_environment_resources(self):
        result = self.beanstalk.describe_environment_resources(environment_name=self.environment)
        self.assertIsInstance(result, response.DescribeEnvironmentResourcesResponse)

    def test_describe_configuration_settings(self):
        result = self.beanstalk.describe_configuration_settings(application_name=self.app_name, environment_name=self.environment)
        self.assertIsInstance(result, response.DescribeConfigurationSettingsResponse)

    def test_request_environment_info(self):
        result = self.beanstalk.request_environment_info(environment_name=self.environment, info_type='tail')
        self.assertIsInstance(result, response.RequestEnvironmentInfoResponse)
        self.wait_for_env(self.environment)
        result = self.beanstalk.retrieve_environment_info(environment_name=self.environment, info_type='tail')
        self.assertIsInstance(result, response.RetrieveEnvironmentInfoResponse)

    def test_rebuild_environment(self):
        result = self.beanstalk.rebuild_environment(environment_name=self.environment)
        self.assertIsInstance(result, response.RebuildEnvironmentResponse)
        self.wait_for_env(self.environment)

    def test_restart_app_server(self):
        result = self.beanstalk.restart_app_server(environment_name=self.environment)
        self.assertIsInstance(result, response.RestartAppServerResponse)
        self.wait_for_env(self.environment)

    def test_update_configuration_template(self):
        result = self.beanstalk.update_configuration_template(application_name=self.app_name, template_name=self.template)
        self.assertIsInstance(result, response.UpdateConfigurationTemplateResponse)

    def test_update_environment(self):
        result = self.beanstalk.update_environment(environment_name=self.environment)
        self.assertIsInstance(result, response.UpdateEnvironmentResponse)
        self.wait_for_env(self.environment)