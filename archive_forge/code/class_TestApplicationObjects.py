import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
class TestApplicationObjects(BasicSuite):

    def create_application(self):
        self.beanstalk.create_application(application_name=self.app_name)
        self.addCleanup(partial(self.beanstalk.delete_application, application_name=self.app_name))

    def test_create_delete_application_version(self):
        app_result = self.beanstalk.create_application(application_name=self.app_name)
        self.assertIsInstance(app_result, response.CreateApplicationResponse)
        self.assertEqual(app_result.application.application_name, self.app_name)
        version_result = self.beanstalk.create_application_version(application_name=self.app_name, version_label=self.app_version)
        self.assertIsInstance(version_result, response.CreateApplicationVersionResponse)
        self.assertEqual(version_result.application_version.version_label, self.app_version)
        result = self.beanstalk.delete_application_version(application_name=self.app_name, version_label=self.app_version)
        self.assertIsInstance(result, response.DeleteApplicationVersionResponse)
        result = self.beanstalk.delete_application(application_name=self.app_name)
        self.assertIsInstance(result, response.DeleteApplicationResponse)

    def test_create_configuration_template(self):
        self.create_application()
        result = self.beanstalk.create_configuration_template(application_name=self.app_name, template_name=self.template, solution_stack_name='32bit Amazon Linux running Tomcat 6')
        self.assertIsInstance(result, response.CreateConfigurationTemplateResponse)
        self.assertEqual(result.solution_stack_name, '32bit Amazon Linux running Tomcat 6')

    def test_create_storage_location(self):
        result = self.beanstalk.create_storage_location()
        self.assertIsInstance(result, response.CreateStorageLocationResponse)

    def test_update_application(self):
        self.create_application()
        result = self.beanstalk.update_application(application_name=self.app_name)
        self.assertIsInstance(result, response.UpdateApplicationResponse)

    def test_update_application_version(self):
        self.create_application()
        self.beanstalk.create_application_version(application_name=self.app_name, version_label=self.app_version)
        result = self.beanstalk.update_application_version(application_name=self.app_name, version_label=self.app_version)
        self.assertIsInstance(result, response.UpdateApplicationVersionResponse)