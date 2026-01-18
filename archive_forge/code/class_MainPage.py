from googleapiclient.discovery import build
import webapp2
from oauth2client.client import GoogleCredentials
class MainPage(webapp2.RequestHandler):

    def get(self):
        self.response.write(get_instances())