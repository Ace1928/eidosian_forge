import os


class Config:
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your_jwt_secret_key")
    # Additional configuration parameters
