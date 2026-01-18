from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, AmbientLight, Vec4, Vec3
from panda3d.core import CollisionTraverser, CollisionNode
from panda3d.core import CollisionHandlerPusher, CollisionSphere
from panda3d.bullet import BulletWorld, BulletBoxShape, BulletRigidBodyNode
from panda3d.bullet import BulletSphereShape
import random
import logging

# Configure logging to the most detailed level possible
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class GameEnvironmentInitializer(ShowBase):
    def __init__(self):
        super().__init__()
        logging.debug("GameEnvironmentInitializer: Superclass initialization complete.")

        self.configureWindowCameraAndBackground()
        self.initializeLightingSystem()
        self.configurePhysicsWorld()
        self.initializeCollisionHandlingSystem()
        self.constructEnvironmentalElements()
        self.scheduleRegularUpdates()
        self.setFrameRate()

    def configureWindowCameraAndBackground(self):
        self.setBackgroundColor(0.1, 0.1, 0.1, 1)
        self.camera.set_pos(0, -50, 20)  # Correct method to set position
        self.camera.look_at(0, 0, 0)  # Correct method to set orientation
        logging.info(
            "GameEnvironmentInitializer: Window, camera, and background color configured."
        )

    def initializeLightingSystem(self):
        self.configureAmbientLight()
        self.configurePointLight()

    def configureAmbientLight(self):
        ambientLight = AmbientLight("ambient_light")
        ambientLight.setColor(Vec4(0.2, 0.2, 0.2, 1))
        ambientLightNode = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNode)
        logging.info("GameEnvironmentInitializer: Ambient light configured.")

    def configurePointLight(self):
        pointLight = PointLight("point_light")
        pointLight.setColor(Vec4(0.9, 0.9, 0.9, 1))
        pointLightNode = self.render.attachNewNode(pointLight)
        pointLightNode.set_pos(10, -20, 20)
        self.render.setLight(pointLightNode)
        logging.info("GameEnvironmentInitializer: Point light configured.")

    def configurePhysicsWorld(self):
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        logging.info(
            "GameEnvironmentInitializer: Physics world configured with gravity."
        )

    def initializeCollisionHandlingSystem(self):
        self.traverser = CollisionTraverser()
        self.pusher = CollisionHandlerPusher()
        logging.info(
            "GameEnvironmentInitializer: Collision handling system initialized."
        )

    def constructEnvironmentalElements(self):
        self.createGround()
        self.createPlayer()
        self.createObstacles()

    def createGround(self):
        shape = BulletBoxShape(Vec3(10, 10, 1))
        body = BulletRigidBodyNode("Ground")
        body.addShape(shape)
        nodePath = self.render.attachNewNode(body)
        nodePath.set_pos(0, 0, -2)
        self.world.attachRigidBody(body)
        logging.debug(
            "GameEnvironmentInitializer: Ground element created and positioned."
        )

    def createPlayer(self):
        shape = BulletSphereShape(1)
        body = BulletRigidBodyNode("Player")
        body.setMass(1.0)
        body.addShape(shape)
        self.playerNodePath = self.render.attachNewNode(body)
        self.playerNodePath.set_pos(0, 0, 2)
        self.world.attachRigidBody(body)
        self.definePlayerCollisionSphere()

    def definePlayerCollisionSphere(self):
        collisionNode = CollisionNode("player")
        collisionNode.addSolid(CollisionSphere(0, 0, 0, 1))
        collisionNodePath = self.playerNodePath.attachNewNode(collisionNode)
        self.traverser.addCollider(collisionNodePath, self.pusher)
        logging.debug("GameEnvironmentInitializer: Player collision sphere defined.")

    def createObstacles(self):
        for _ in range(10):
            x, y, z = random.uniform(-8, 8), random.uniform(-8, 8), 0
            shape = BulletBoxShape(Vec3(1, 1, 1))
            body = BulletRigidBodyNode("Obstacle")
            body.addShape(shape)
            nodePath = self.render.attachNewNode(body)
            nodePath.set_pos(x, y, z)
            self.world.attachRigidBody(body)
        logging.debug("GameEnvironmentInitializer: Obstacles created and positioned.")

    def scheduleRegularUpdates(self):
        self.taskManager.add(self.updatePhysicsAndLogging, "update")
        logging.info("GameEnvironmentInitializer: Regular updates scheduled.")

    def setFrameRate(self):
        globalClock.setFrameRate(60)
        logging.info("GameEnvironmentInitializer: Frame rate set to 60 FPS.")

    def updatePhysicsAndLogging(self, task):
        deltaTime = globalClock.getDeltaTime()
        self.world.doPhysics(deltaTime)
        playerPosition = self.playerNodePath.getPosition()
        logging.debug(
            f"GameEnvironmentInitializer: Physics updated for deltaTime: {deltaTime}, Player position: {playerPosition}"
        )
        return task.cont


gameEnvironment = GameEnvironmentInitializer()
gameEnvironment.run()
logging.info(
    "GameEnvironmentInitializer: Game execution started and main loop running."
)
